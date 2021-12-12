#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include <hdf5.h>
#include <assert.h>

#include "../include/nuhf.h"

struct cell {
    double x[3];
    double w;
    int level;
    int rel_x[3]; //position within parent cell
    int refined;
    struct cell *children[8];
    struct cell *parent;
    
    double value;
    
    int label;
};

#define wrap(i,N) ((i)%(N)+(N))%(N)
#define fwrap(x,L) fmod(fmod((x),(L))+(L),(L))

static inline int row_major(int i, int j, int k, int N) {
    i = wrap(i,N);
    j = wrap(j,N);
    k = wrap(k,N);
    return i*N*N + j*N + k;
}

int writeFieldFile(const double *box, int N, double boxlen, const char *fname) {
    /* Create the hdf5 file */
    hid_t h_file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    /* Create the Header group */
    hid_t h_grp = H5Gcreate(h_file, "/Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for BoxSize attribute */
    const hsize_t arank = 1;
    const hsize_t adims[1] = {3}; //3D space
    hid_t h_aspace = H5Screate_simple(arank, adims, NULL);

    /* Create the BoxSize attribute and write the data */
    hid_t h_attr = H5Acreate1(h_grp, "BoxSize", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    double boxsize[3] = {boxlen, boxlen, boxlen};
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, boxsize);

    /* Close the attribute, corresponding dataspace, and the Header group */
    H5Aclose(h_attr);
    H5Sclose(h_aspace);
    H5Gclose(h_grp);

    /* Create the Field group */
    h_grp = H5Gcreate(h_file, "/Field", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for the field */
    const hsize_t frank = 3;
    const hsize_t fdims[3] = {N, N, N}; //3D space
    hid_t h_fspace = H5Screate_simple(frank, fdims, NULL);

    /* Create the dataset for the field */
    hid_t h_data = H5Dcreate(h_grp, "Field", H5T_NATIVE_DOUBLE, h_fspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Write the data */
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_fspace, h_fspace, H5P_DEFAULT, box);

    /* Close the dataset, corresponding dataspace, and the Field group */
    H5Dclose(h_data);
    H5Sclose(h_fspace);
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_file);

    return 0;
}

/* Determine whether a cell has a neighbour in the (i,j,k) direction at the
 * same level of refinement.
 *
 * @param c The cell
 * @param n The neighbour (if it exists)
 * @param i Distance in the x[0]-direction (i = -1, 0, 1)
 * @param j Distance in the x[1]-direction (i = -1, 0, 1)
 * @param k Distance in the x[2]-direction (i = -1, 0, 1)
 * @param N Dimension of the domain (coarsest level) grid 
 * @param domain The domain grid
 * 
 * Returns 1 if the neighbour exists, 0 if it does not
 */
int has_neighbour(struct cell *c, struct cell **n, int i, int j, int k, int N, struct cell *domain) {
    if (i < -1 || i > 1 || j < -1 || j > 1 || k < -1 || k > 1) {
        printf("Error: searching for non-adjacent neighbour.\n");
        return 0;
    }
        
    /* Top-level domain search */
    if (c->level == 0) {
        int x = c->rel_x[0] + i;
        int y = c->rel_x[1] + j;
        int z = c->rel_x[2] + k;
        
        
        /* Neighbours always exist on the domain */
        *n = &domain[row_major(x, y, z, N)];
        return 1;
    } else {
        int x = c->rel_x[0] + i;
        int y = c->rel_x[1] + j;
        int z = c->rel_x[2] + k;
        
        /* The neighbour is a sibling */
        if (x >= 0 && x < 2 && y >= 0 && y < 2 && z >= 0 && z < 2) {
            *n = c->parent->children[row_major(x, y, z, 2)];
            return 1;
        }
            
        /* Move up a parent */
        int px = (x == -1) ? -1 : ((x == 2) ? 1 : 0);
        int py = (y == -1) ? -1 : ((y == 2) ? 1 : 0);
        int pz = (z == -1) ? -1 : ((z == 2) ? 1 : 0);
                
        /* Does the parent have a neighbour? */
        struct cell *pn;
        int has_pn = has_neighbour(c->parent, &pn, px, py, pz, N, domain);
        
        /* The target cell has a neighbour if the parent has a neighbour and
         * the parent's neighbour is refined. */
        if (has_pn && pn->refined) {
            /* Find the relative position of the target's neighbour inside the
             * parent's neighbour */
            x -= px * 2;
            y -= py * 2;
            z -= pz * 2;
        
            if (x >= 0 && x < 2 && y >= 0 && y < 2 && z >= 0 && z < 2) {
                *n = pn->children[row_major(x, y, z, 2)];
                return 1;
            } else {
                printf("Logic error.");
                return 0;
            }
        } else {
            return 0;
        }
    }
}

int find(int *labels, int x)  {
    int y = x;
    while (labels[y] != y)
        y = labels[y];
    while (labels[x] != x)  {
        int z = labels[x];
        labels[x] = y;
        x = z;
    }
    return y;
}

void unite(int *labels, int x, int y)  {
    labels[find(labels, x)] = find(labels, y);
}

void refine_cell(struct cell *c, struct cell *refinements, int *refinement_counter, int refinement_memory) {
    /* This is cell is now refined */
    c->refined = 1;
    
    /* Split the cell in 8 sub-cells */
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {         
                
                if (*refinement_counter > refinement_memory) {
                    printf("We need more memory for refinements! %d %d\n", *refinement_counter, refinement_memory);
                    exit(1);
                }
                
                refinements[*refinement_counter].parent = c;
                refinements[*refinement_counter].x[0] = c->x[0] + i * 0.5 * c->w;
                refinements[*refinement_counter].x[1] = c->x[1] + j * 0.5 * c->w;
                refinements[*refinement_counter].x[2] = c->x[2] + k * 0.5 * c->w;
                refinements[*refinement_counter].w = 0.5 * c->w;
                refinements[*refinement_counter].level = c->level + 1;
                refinements[*refinement_counter].refined = 0;
                refinements[*refinement_counter].rel_x[0] = i;
                refinements[*refinement_counter].rel_x[1] = j;
                refinements[*refinement_counter].rel_x[2] = k;
                refinements[*refinement_counter].value = 0.0;
                
                c->children[row_major(i, j, k, 2)] = &refinements[*refinement_counter];
                *refinement_counter = *refinement_counter + 1;
            }
        }
    }
}

struct pair {
    int idx;
    int xyz;
};

/* Sort by row_major(x,y,z) */
static inline int compareByVal(const void *a, const void *b) {
    struct pair *ca = (struct pair*) a;
    struct pair *cb = (struct pair*) b;
    return ca->xyz >= cb->xyz;
}


/* Is a given point contained in a cell at a certain refinement level? */
int cell_exists(int level, double x, double y, double z, double DomainRes, int N, int start, struct cell **c, struct cell *domain, struct cell *refinements) {
    /* Locate the cell in the domain grid */
    if (start) {
        double X = x / DomainRes;
        double Y = y / DomainRes;
        double Z = z / DomainRes;
        int iX = (int) floor(X);
        int iY = (int) floor(Y);
        int iZ = (int) floor(Z);
        
        struct cell *dc = &domain[row_major(iX, iY, iZ, N)];
        *c = dc;
        cell_exists(level, x, y, z, DomainRes, N, 0, c, domain, refinements);
    } else if ((*c)->level < level && (*c)->refined) {
        int scale = 2 << (*c)->level;
        double Res = DomainRes / scale;
        double X = (x - (*c)->x[0]) / Res;
        double Y = (y - (*c)->x[1]) / Res;
        double Z = (z - (*c)->x[2]) / Res;
        int iX = (int) floor(X);
        int iY = (int) floor(Y);
        int iZ = (int) floor(Z);
    
        struct cell *child = (*c)->children[row_major(iX, iY, iZ, 2)];
        *c = child;  
        cell_exists(level, x, y, z, DomainRes, N, 0, c, domain, refinements);     
    } else if ((*c)->level == level) {
        return 1;
    } else {
        return 0;
    }
    return 0;
}

int find_top_cell(int level, int x, int y, int z, struct cell **c, int N, struct cell *domain) {
    int scale = 2 << (level - 1);
    int ix = x / scale;
    int iy = y / scale;
    int iz = z / scale;
    *c = &domain[row_major(ix, iy, iz, N)];
    return 1;
}

int find_cell(int level, int x, int y, int z, struct cell **c) {
    if ((*c)->level == level) {
        return 1;
    } else if (!(*c)->refined) {
        return 0;
    } else {
        int level_diff = level - (*c)->level;
        int scale = 2 << (level_diff - 1);
        double inv_w = 1.0 / (*c)->w;
        double inv_scale = 1.0 / scale;
        int cx = (*c)->x[0] * inv_w;
        int cy = (*c)->x[1] * inv_w;
        int cz = (*c)->x[2] * inv_w;
        double hx = x * inv_scale - cx;
        double hy = y * inv_scale - cy;
        double hz = z * inv_scale - cz;
        int rx = 2 * hx;
        int ry = 2 * hy;
        int rz = 2 * hz;
        *c = (*c)->children[row_major(rx, ry, rz, 2)];
        return find_cell(level, x, y, z, c);
    }
}

void deposit_tsc(double value, double x, double y, double z, int level, struct cell *domain, int N, double DomainRes) {
    int scale = 2 << (level - 1);
    double res = DomainRes / scale;
    double inv_res = 1.0 / res;
    double X = x * inv_res;
    double Y = y * inv_res;
    double Z = z * inv_res;
    int iX = (int) floor(X);
    int iY = (int) floor(Y);
    int iZ = (int) floor(Z);
    
    double grid_cell_vol = res * res * res;
    
    /* The search window with respect to the top-left-upper corner */
	int lookLftX = (int) floor((X-iX) - 1.5);
	int lookRgtX = (int) floor((X-iX) + 1.5);
	int lookLftY = (int) floor((Y-iY) - 1.5);
	int lookRgtY = (int) floor((Y-iY) + 1.5);
	int lookLftZ = (int) floor((Z-iZ) - 1.5);
	int lookRgtZ = (int) floor((Z-iZ) + 1.5);
    
    /* Do the mass assignment */
	for (int i=lookLftX; i<=lookRgtX; i++) {
		for (int j=lookLftY; j<=lookRgtY; j++) {
			for (int k=lookLftZ; k<=lookRgtZ; k++) {
                double xx = fabs(X - (iX+i));
                double yy = fabs(Y - (iY+j));
                double zz = fabs(Z - (iZ+k));

                double part_x = xx < 0.5 ? (0.75-xx*xx)
                                        : (xx < 1.5 ? 0.5*(1.5-xx)*(1.5-xx) : 0);
				double part_y = yy < 0.5 ? (0.75-yy*yy)
                                        : (yy < 1.5 ? 0.5*(1.5-yy)*(1.5-yy) : 0);
				double part_z = zz < 0.5 ? (0.75-zz*zz)
                                        : (zz < 1.5 ? 0.5*(1.5-zz)*(1.5-zz) : 0);
                                        
                if (part_x * part_y * part_z == 0.) continue;
                                        
                /* Find the cell in the refined grid (if it exists) */
                struct cell *c;
                find_top_cell(level, iX+i, iY+j, iZ+k, &c, N, domain);
                int exists = find_cell(level, iX+i, iY+j, iZ+k, &c);
                if (exists) {
                    c->value += value / grid_cell_vol * (part_x*part_y*part_z);
                    // printf("deposited %g at %g %g %g\n", value / grid_cell_vol * (part_x*part_y*part_z), c->x[0], c->x[1], c->x[2]);
                }
			}
		}
	}
    
}

int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("No parameter file specified.\n");
        return 0;
    }
    
    printf("nuHF halo finder\n");
    
    /* Initialize MPI for distributed memory parallelization */
    MPI_Init(&argc, &argv);
    // fftw_mpi_init();

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);
    
    /* Read options */
    const char *fname = argv[1];
    message(rank, "The parameter file is %s\n", fname);

    struct params pars;
    struct units us;
    struct particle_type *types = NULL;
    struct cosmology cosmo;
    
    /* Read parameter file */
    readParams(&pars, fname);
    readUnits(&us, fname);
    readCosmology(&cosmo, &us, fname);
    // readTypes(&pars, &types, fname);
    
    /* Option to override the input filename by specifying command line option */
    if (argc > 2) {
        const char *input_filename = argv[2];
        strcpy(pars.InputFilename, input_filename);
    }
    message(rank, "Reading simulation snapshot from: \"%s\".\n", pars.InputFilename);
    
    /* Open the file */
    // hid_t h_file = openFile_MPI(MPI_COMM_WORLD, pars.InputFilename);
    hid_t h_file = H5Fopen(pars.InputFilename, H5F_ACC_RDONLY, H5P_DEFAULT);

    /* Open the Header group */
    hid_t h_grp = H5Gopen(h_file, "Header", H5P_DEFAULT);

    /* Read the physical dimensions of the box */
    double boxlen[3];
    hid_t h_attr = H5Aopen(h_grp, "BoxSize", H5P_DEFAULT);
    hid_t h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, boxlen);
    H5Aclose(h_attr);
    assert(h_err >= 0);

    const double BoxLen = boxlen[0];
    message(rank, "Reading particle type '%s'.\n", pars.ImportName);
    message(rank, "BoxSize is %g\n", BoxLen);

    /* Read the numbers of particles of each type */
    hsize_t numer_of_types;
    h_attr = H5Aopen(h_grp, "NumPart_Total", H5P_DEFAULT);
    hid_t h_atspace = H5Aget_space(h_attr);
    H5Sget_simple_extent_dims(h_atspace, &numer_of_types, NULL);
    H5Sclose(h_atspace);
    H5Aclose(h_attr);

    /* Close the Header group again */
    H5Gclose(h_grp);

    /* Check if the Cosmology group exists */
    hid_t h_status = H5Eset_auto1(NULL, NULL);  //turn off error printing
    h_status = H5Gget_objinfo(h_file, "/Cosmology", 0, NULL);

    /* If the group exists. */
    if (h_status == 0) {
        /* Open the Cosmology group */
        h_grp = H5Gopen(h_file, "Cosmology", H5P_DEFAULT);

        /* Read the redshift attribute */
        double redshift;
        h_attr = H5Aopen(h_grp, "Redshift", H5P_DEFAULT);
        h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, &redshift);
        H5Aclose(h_attr);
        assert(h_err >= 0);

        message(rank, "The redshift was %f\n\n", redshift);

        /* Close the Cosmology group */
        H5Gclose(h_grp);
    }
    
    
    /* Open the corresponding group */
    h_grp = H5Gopen(h_file, pars.ImportName, H5P_DEFAULT);

    /* Open the coordinates dataset */
    hid_t h_dat = H5Dopen(h_grp, "Coordinates", H5P_DEFAULT);

    /* Find the dataspace (in the file) */
    hid_t h_space = H5Dget_space (h_dat);

    /* Get the dimensions of this dataspace */
    hsize_t dims[2];
    H5Sget_simple_extent_dims(h_space, dims, NULL);

    /* How many particles do we want per slab? */
    hid_t Npart = dims[0];
    hid_t max_slab_size = pars.SlabSize;
    int slabs = Npart/max_slab_size;
    hid_t counter = 0;

    /* Close the data and memory spaces */
    H5Sclose(h_space);

    /* Close the dataset */
    H5Dclose(h_dat);

    double total_mass = 0; //for this particle type

    int slab_counter = 0;

    message(rank, "\n");
        
    printf("The GridSize is %d\n", pars.GridSize);
    
    const int N = pars.GridSize;
    const double DomainRes = BoxLen / N;
    struct cell *domain = calloc(sizeof(struct cell), N * N * N);
    
    /* Create the domain (coarsest level) grid */
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            for (int z = 0; z < N; z++) {
                domain[row_major(x, y, z, N)].x[0] = DomainRes * x;
                domain[row_major(x, y, z, N)].x[1] = DomainRes * y;
                domain[row_major(x, y, z, N)].x[2] = DomainRes * z;
                domain[row_major(x, y, z, N)].w = DomainRes;
                domain[row_major(x, y, z, N)].level = 0;
                domain[row_major(x, y, z, N)].refined = 0;
                domain[row_major(x, y, z, N)].rel_x[0] = x;
                domain[row_major(x, y, z, N)].rel_x[1] = y;
                domain[row_major(x, y, z, N)].rel_x[2] = z;
                domain[row_major(x, y, z, N)].value = 0.0;
                
                // printf ("%g %g %g\n", domain[row_major(x, y, z, N)].x[0], domain[row_major(x, y, z, N)].x[1], domain[row_major(x, y, z, N)].x[2]);
            }
        }
    }
    
    printf("=====\n");
    
    for (int k=rank; k<slabs+1; k+=MPI_Rank_Count) {
        /* All slabs have the same number of particles, except possibly the last */
        hid_t slab_size = fmin(Npart - k * max_slab_size, max_slab_size);
        counter += slab_size; //the number of particles read

        /* Define the hyperslab */
        hsize_t slab_dims[2], start[2]; //for 3-vectors
        hsize_t slab_dims_one[1], start_one[1]; //for scalars

        /* Slab dimensions for 3-vectors */
        slab_dims[0] = slab_size;
        slab_dims[1] = 3; //(x,y,z)
        start[0] = k * max_slab_size;
        start[1] = 0; //start with x

        /* Slab dimensions for scalars */
        slab_dims_one[0] = slab_size;
        start_one[0] = k * max_slab_size;

        /* Open the coordinates dataset */
        h_dat = H5Dopen(h_grp, "Coordinates", H5P_DEFAULT);

        /* Find the dataspace (in the file) */
        h_space = H5Dget_space (h_dat);

        /* Select the hyperslab */
        hid_t status = H5Sselect_hyperslab(h_space, H5S_SELECT_SET, start,
                                           NULL, slab_dims, NULL);
        assert(status >= 0);

        /* Create a memory space */
        hid_t h_mems = H5Screate_simple(2, slab_dims, NULL);

        /* Create the data array */
        double data[slab_size][3];

        status = H5Dread(h_dat, H5T_NATIVE_DOUBLE, h_mems, h_space, H5P_DEFAULT,
                         data);

        /* Close the memory space */
        H5Sclose(h_mems);

        /* Close the data and memory spaces */
        H5Sclose(h_space);

        /* Close the dataset */
        H5Dclose(h_dat);


        /* Open the masses dataset */
        h_dat = H5Dopen(h_grp, "Masses", H5P_DEFAULT);

        /* Find the dataspace (in the file) */
        h_space = H5Dget_space (h_dat);

        /* Select the hyperslab */
        status = H5Sselect_hyperslab(h_space, H5S_SELECT_SET, start_one, NULL,
                                            slab_dims_one, NULL);

        /* Create a memory space */
        h_mems = H5Screate_simple(1, slab_dims_one, NULL);

        /* Create the data array */
        double mass_data[slab_size];

        status = H5Dread(h_dat, H5T_NATIVE_DOUBLE, h_mems, h_space, H5P_DEFAULT,
                         mass_data);

        /* Close the memory space */
        H5Sclose(h_mems);

        /* Close the data and memory spaces */
        H5Sclose(h_space);

        /* Close the dataset */
        H5Dclose(h_dat);

        double grid_cell_vol = DomainRes * DomainRes * DomainRes;

        /* Assign the particles to the grid with CIC */
        for (int l=0; l<slab_size; l++) {
            double X = data[l][0] / DomainRes;
            double Y = data[l][1] / DomainRes;
            double Z = data[l][2] / DomainRes;

            double M = mass_data[l];
            total_mass += M;

            int iX = (int) floor(X);
            int iY = (int) floor(Y);
            int iZ = (int) floor(Z);

            //The search window with respect to the top-left-upper corner
    		int lookLftX = (int) floor((X-iX) - 1.5);
    		int lookRgtX = (int) floor((X-iX) + 1.5);
    		int lookLftY = (int) floor((Y-iY) - 1.5);
    		int lookRgtY = (int) floor((Y-iY) + 1.5);
    		int lookLftZ = (int) floor((Z-iZ) - 1.5);
    		int lookRgtZ = (int) floor((Z-iZ) + 1.5);
        
            //Do the mass assignment
    		for (int x=lookLftX; x<=lookRgtX; x++) {
    			for (int y=lookLftY; y<=lookRgtY; y++) {
    				for (int z=lookLftZ; z<=lookRgtZ; z++) {
                        double xx = fabs(X - (iX+x));
                        double yy = fabs(Y - (iY+y));
                        double zz = fabs(Z - (iZ+z));
        
                        double part_x = xx < 0.5 ? (0.75-xx*xx)
                                                : (xx < 1.5 ? 0.5*(1.5-xx)*(1.5-xx) : 0);
        				double part_y = yy < 0.5 ? (0.75-yy*yy)
                                                : (yy < 1.5 ? 0.5*(1.5-yy)*(1.5-yy) : 0);
        				double part_z = zz < 0.5 ? (0.75-zz*zz)
                                                : (zz < 1.5 ? 0.5*(1.5-zz)*(1.5-zz) : 0);
        
                        domain[row_major(iX+x, iY+y, iZ+z, N)].value += M/grid_cell_vol * (part_x*part_y*part_z);
    				}
    			}
    		}
        }

        printf("(%03d,%03d) Read %ld particles\n", rank, k, slab_size);
        slab_counter++;
    }
    
    /* Turn it into an overdensity grid */
    double avg_density = total_mass / (BoxLen * BoxLen * BoxLen);
    
    for (int i=0; i<N*N*N; i++) {
        domain[i].value = (domain[i].value - avg_density) / avg_density;
    }
    
    double *box = malloc(sizeof(double) * N * N * N);
    for (int i=0; i<N*N*N; i++) {
        box[i] = domain[i].value;
    }
    writeFieldFile(box, N, BoxLen, "density.hdf5");
    free(box);
    
    printf("=====\n");
    printf("Doing the first refinement.\n");
    printf("=====\n");
    
    const double density_criterion = 500;
    const double density_criterion2 = density_criterion;
    
    int refinement_memory = 10 * N * N * N;
    int refinement_counter = 0;
    struct cell *refinements = calloc(sizeof(struct cell), refinement_memory); 
    
    /* Refine the octree */
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            for (int z = 0; z < N; z++) {
                
                /* Refine? */
                int refine = domain[row_major(x, y, z, N)].value > density_criterion;
                
                if (refine) {
                    printf("Refining %d %d %d\n", x, y, z);
                    
                    refine_cell(&domain[row_major(x, y, z, N)], refinements, &refinement_counter, refinement_memory);
                }
            }
        }
    }
    
    /* Do mass assignment on the refined level */
    counter = 0;
    for (int k=rank; k<slabs+1; k+=MPI_Rank_Count) {
        /* All slabs have the same number of particles, except possibly the last */
        hid_t slab_size = fmin(Npart - k * max_slab_size, max_slab_size);
        counter += slab_size; //the number of particles read

        /* Define the hyperslab */
        hsize_t slab_dims[2], start[2]; //for 3-vectors
        hsize_t slab_dims_one[1], start_one[1]; //for scalars

        /* Slab dimensions for 3-vectors */
        slab_dims[0] = slab_size;
        slab_dims[1] = 3; //(x,y,z)
        start[0] = k * max_slab_size;
        start[1] = 0; //start with x

        /* Slab dimensions for scalars */
        slab_dims_one[0] = slab_size;
        start_one[0] = k * max_slab_size;

        /* Open the coordinates dataset */
        h_dat = H5Dopen(h_grp, "Coordinates", H5P_DEFAULT);

        /* Find the dataspace (in the file) */
        h_space = H5Dget_space (h_dat);

        /* Select the hyperslab */
        hid_t status = H5Sselect_hyperslab(h_space, H5S_SELECT_SET, start,
                                           NULL, slab_dims, NULL);
        assert(status >= 0);

        /* Create a memory space */
        hid_t h_mems = H5Screate_simple(2, slab_dims, NULL);

        /* Create the data array */
        double data[slab_size][3];

        status = H5Dread(h_dat, H5T_NATIVE_DOUBLE, h_mems, h_space, H5P_DEFAULT,
                         data);

        /* Close the memory space */
        H5Sclose(h_mems);

        /* Close the data and memory spaces */
        H5Sclose(h_space);

        /* Close the dataset */
        H5Dclose(h_dat);


        /* Open the masses dataset */
        h_dat = H5Dopen(h_grp, "Masses", H5P_DEFAULT);

        /* Find the dataspace (in the file) */
        h_space = H5Dget_space (h_dat);

        /* Select the hyperslab */
        status = H5Sselect_hyperslab(h_space, H5S_SELECT_SET, start_one, NULL,
                                            slab_dims_one, NULL);

        /* Create a memory space */
        h_mems = H5Screate_simple(1, slab_dims_one, NULL);

        /* Create the data array */
        double mass_data[slab_size];

        status = H5Dread(h_dat, H5T_NATIVE_DOUBLE, h_mems, h_space, H5P_DEFAULT,
                         mass_data);

        /* Close the memory space */
        H5Sclose(h_mems);

        /* Close the data and memory spaces */
        H5Sclose(h_space);

        /* Close the dataset */
        H5Dclose(h_dat);

        /* Assign the particles to the grid with CIC */
        for (int l=0; l<slab_size; l++) {
            double X = data[l][0];
            double Y = data[l][1];
            double Z = data[l][2];
            double M = mass_data[l];
            
            deposit_tsc(M, X, Y, Z, 1, domain, N, DomainRes);
        }

        printf("(%03d,%03d) Read %ld particles\n", rank, k, slab_size);
        slab_counter++;
    }
    
    
    printf("=====\n");
    
    /* Turn it into an overdensity grid */
    double refined_avg_density = total_mass / (BoxLen * BoxLen * BoxLen) / 8;
    
    for (int i = 0; i < refinement_counter; i++) {
        refinements[i].value = (refinements[i].value - refined_avg_density) / refined_avg_density;
    }
    
    printf("Avg density %g %g\n", avg_density, refined_avg_density);
        
    box = calloc(sizeof(double), 8 * N * N * N);
    for (int i = 0; i < refinement_counter; i++) {
        struct cell *c = &refinements[i];
        int x = c->x[0] / c->w;
        int y = c->x[1] / c->w;
        int z = c->x[2] / c->w;
        int level = c->level;
        int scale = 2 << (level - 1);
        int xyz = row_major(x, y, z, N * scale);
        
        box[xyz] = c->value;
    }
    writeFieldFile(box, 2*N, BoxLen, "density2.hdf5");
    free(box);    
    
    printf("=====\n");
    printf("Doing the second refinement.\n");
    printf("=====\n");
    
    /* Further refinement? */
    for (int i = 0; i < refinement_counter; i++) {
        /* Refine? */
        int refine = refinements[i].value > density_criterion2 * 8;
            
        if (refine) {
            printf("Refining %g %g %g %g\n", refinements[i].x[0], refinements[i].x[1], refinements[i].x[2], refinements[i].value);
            
            refine_cell(&refinements[i], refinements, &refinement_counter, refinement_memory);
        }
    }
    
    /* Do mass assignment on the refined level */
    counter = 0;
    for (int k=rank; k<slabs+1; k+=MPI_Rank_Count) {
        /* All slabs have the same number of particles, except possibly the last */
        hid_t slab_size = fmin(Npart - k * max_slab_size, max_slab_size);
        counter += slab_size; //the number of particles read

        /* Define the hyperslab */
        hsize_t slab_dims[2], start[2]; //for 3-vectors
        hsize_t slab_dims_one[1], start_one[1]; //for scalars

        /* Slab dimensions for 3-vectors */
        slab_dims[0] = slab_size;
        slab_dims[1] = 3; //(x,y,z)
        start[0] = k * max_slab_size;
        start[1] = 0; //start with x

        /* Slab dimensions for scalars */
        slab_dims_one[0] = slab_size;
        start_one[0] = k * max_slab_size;

        /* Open the coordinates dataset */
        h_dat = H5Dopen(h_grp, "Coordinates", H5P_DEFAULT);

        /* Find the dataspace (in the file) */
        h_space = H5Dget_space (h_dat);

        /* Select the hyperslab */
        hid_t status = H5Sselect_hyperslab(h_space, H5S_SELECT_SET, start,
                                           NULL, slab_dims, NULL);
        assert(status >= 0);

        /* Create a memory space */
        hid_t h_mems = H5Screate_simple(2, slab_dims, NULL);

        /* Create the data array */
        double data[slab_size][3];

        status = H5Dread(h_dat, H5T_NATIVE_DOUBLE, h_mems, h_space, H5P_DEFAULT,
                         data);

        /* Close the memory space */
        H5Sclose(h_mems);

        /* Close the data and memory spaces */
        H5Sclose(h_space);

        /* Close the dataset */
        H5Dclose(h_dat);


        /* Open the masses dataset */
        h_dat = H5Dopen(h_grp, "Masses", H5P_DEFAULT);

        /* Find the dataspace (in the file) */
        h_space = H5Dget_space (h_dat);

        /* Select the hyperslab */
        status = H5Sselect_hyperslab(h_space, H5S_SELECT_SET, start_one, NULL,
                                            slab_dims_one, NULL);

        /* Create a memory space */
        h_mems = H5Screate_simple(1, slab_dims_one, NULL);

        /* Create the data array */
        double mass_data[slab_size];

        status = H5Dread(h_dat, H5T_NATIVE_DOUBLE, h_mems, h_space, H5P_DEFAULT,
                         mass_data);

        /* Close the memory space */
        H5Sclose(h_mems);

        /* Close the data and memory spaces */
        H5Sclose(h_space);

        /* Close the dataset */
        H5Dclose(h_dat);

        /* Assign the particles to the grid with CIC */
        for (int l=0; l<slab_size; l++) {
            double X = data[l][0];
            double Y = data[l][1];
            double Z = data[l][2];
            double M = mass_data[l];
            
            deposit_tsc(M, X, Y, Z, 2, domain, N, DomainRes);
        }

        printf("(%03d,%03d) Read %ld particles\n", rank, k, slab_size);
        slab_counter++;
    }
    
    
    printf("=====\n");
    
    /* Turn it into an overdensity grid */
    double refined_avg_density2 = total_mass / (BoxLen * BoxLen * BoxLen) / 64;
    
    for (int i = 0; i < refinement_counter; i++) {
        if (refinements[i].level == 2) {
            refinements[i].value = (refinements[i].value - refined_avg_density2) / refined_avg_density2;
        }
    }
    
    printf("Avg density %g %g %g\n", avg_density, refined_avg_density, refined_avg_density2);
        
    box = calloc(sizeof(double), 64 * N * N * N);
    for (int i = 0; i < refinement_counter; i++) {
        if (refinements[i].level == 2) {
            struct cell *c = &refinements[i];
            int x = c->x[0] / c->w;
            int y = c->x[1] / c->w;
            int z = c->x[2] / c->w;
            int level = c->level;
            int scale = 2 << (level - 1);
            int xyz = row_major(x, y, z, N * scale);
            
            box[xyz] = c->value;
        }
    }
    writeFieldFile(box, 4*N, BoxLen, "density3.hdf5");
    free(box);    
    
    printf("=====\n");
    printf("Doing the third refinement.\n");
    printf("=====\n");
    
    
    /* Further refinement? */
    for (int i = 0; i < refinement_counter; i++) {
        if (refinements[i].level == 2) {
            /* Refine? */
            int refine = refinements[i].value > density_criterion2 * 64;
                
            if (refine) {
                printf("Refining %g %g %g %g\n", refinements[i].x[0], refinements[i].x[1], refinements[i].x[2], refinements[i].value);
                
                refine_cell(&refinements[i], refinements, &refinement_counter, refinement_memory);
            }
        }
    }
    
    exit(1);
    
    
    int max_level = 1;
    
    /* Find isolated regions at a specific level using Hoshen-Kopelman */
    int begin = 0;
    int end = refinement_counter;
        
    int *labels = malloc(sizeof(int) * (end - begin));
    for (int i = 0; i < end - begin; i++) {
        labels[i] = i;
    }
    
    /* Set the labels to zero */
    for (int i = begin; i < end; i++) {
        struct cell *c = &refinements[i];
        c->label = 0;
    }
    
    int largest_label = 0;
    
    /* Sort the cells at this level by their position in a grid scan */
    struct pair *positions = malloc(sizeof(struct pair) * (end - begin));
    for (int i = begin; i < end; i++) {
        positions[i - begin].idx = i - begin;
        struct cell *c = &refinements[i];
        int x = c->x[0] / c->w;
        int y = c->x[1] / c->w;
        int z = c->x[2] / c->w;
        int level = c->level;
        int scale = 2 << (level - 1);
        int xyz = row_major(x, y, z, N * scale);
        positions[i - begin].xyz = xyz;
        printf("%d %d\n", i - begin, xyz);
    }    
    
    /* Sort the indices */
    qsort(positions, end - begin, sizeof(struct pair), compareByVal);
        
    printf("===\n");
    for (int i = begin; i < end; i++) {
        printf("%d %d\n", positions[i - begin].idx, positions[i - begin].xyz);
    }
    printf("===\n");
    
    
    /* Do a scan at the required level */
    for (int i = 0; i < end - begin; i++) {
        struct cell *c = &refinements[positions[i].idx];
        printf ("%g %g %g\n", c->x[0], c->x[1], c->x[2]);
        
        
        
        struct cell *left, *top, *up;
        int has_left = has_neighbour(c, &left, -1, 0, 0, N, domain);
        int has_top = has_neighbour(c, &top, 0, -1, 0, N, domain);
        int has_up = has_neighbour(c, &up, 0, 0, -1, N, domain);
        
        // printf("n: %d %d %d\n", has_left, has_top, has_up);
        
        if (!has_left && !has_top && !has_up) { /* Neither a label above nor to the left. */
            largest_label = largest_label + 1; /* Make a new, as-yet-unused cluster label. */
            c->label = largest_label;
        } else if (has_left && !has_top && !has_up) { /* One neighbor, to the left. */
            c->label = find(labels, left->label);
        } else if (!has_left && has_top && !has_up) { /* One neighbor, above. */
            c->label = find(labels, top->label);
        } else if (!has_left && !has_top && has_up) { /* One neighbor, up. */
            c->label = find(labels, up->label);
        } else if (has_left && has_top && !has_up) { /* Neighbours left and top */
            unite(labels, left->label, top->label); /* Link the left and above clusters. */
            c->label = find(labels, left->label);
        } else if (has_left && !has_top && has_up) { /* Neighbours left and up */
            unite(labels, left->label, up->label); /* Link the left and up clusters. */
            c->label = find(labels, left->label);
        } else if (!has_left && has_top && has_up) { /* Neighbours top and up */
            unite(labels, top->label, up->label); /* Link the top and up clusters. */
            c->label = find(labels, top->label);
        } else if (has_left && has_top && has_up) { /* Neighbours left, top, and up */
            unite(labels, left->label, top->label); /* Link the left and top clusters. */
            unite(labels, left->label, up->label); /* Link the left and up clusters. */
            c->label = find(labels, left->label);
        }
    }
    
    printf("===\n");
    
    for (int i = 0; i < end - begin; i++) {
        struct cell *c = &refinements[positions[i].idx];
        printf ("%g %g %g %d\n", c->x[0] / c->w, c->x[1] / c->w, c->x[2] / c->w, c->label);
    }
    
    
    /* Can we do neighbour finding? */
    // domain[row_major(x, y, z, N)]
    struct cell *c = &refinements[3];
    struct cell *n;
    int has = has_neighbour(c, &n, -1, 0, 0, N, domain);
    
    printf("Neuighbour? %d\n", has);
    printf("%d %d %d %d\n", c->rel_x[0], c->rel_x[1], c->rel_x[2], c->level);
    if (has)
    printf("%d %d %d %d\n", n->rel_x[0], n->rel_x[1], n->rel_x[2], n->level);
    
    
    struct cell *c2;
    int ex = cell_exists(1, 1.5, 3.5, 3.5, DomainRes, N, 1, &c2, domain, refinements);
    printf("exists? %d\n", ex);
    printf("exists? %g %g %g\n", c2->x[0], c2->x[1], c2->x[2]);
    
    
    double X = 7.7;
    double Y = 3.0;
    double Z = 3.0;
    int ix = X / (DomainRes / 2);
    int iy = Y / (DomainRes / 2);
    int iz = Z / (DomainRes / 2);
    
    struct cell *tc;
    find_top_cell(1, ix, iy, iz, &tc, N, domain);
    printf("found %g %g %g %d\n", tc->x[0], tc->x[1], tc->x[2], tc->level);
    int found = find_cell(1, ix, iy, iz, &tc);
    
    printf("found %d %d %d %d\n", found, ix, iy, iz);
    printf("found %d %g %g %g %d\n", found, tc->x[0], tc->x[1], tc->x[2], tc->level);
    
    
    /* Deposit some particles */
    double mass = 1.0;
    deposit_tsc(mass, X, Y, Z, 1, domain, N, DomainRes);
    
    
    
    
    free(refinements);
    free(domain);
    
    /* Done with MPI parallelization */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    /* Clean up */
    // cleanTypes(&pars, &types);
    cleanParams(&pars);
    
    
    return 0;
}