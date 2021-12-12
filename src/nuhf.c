#include <stdio.h>
#include <stdlib.h>
#include <math.h>


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
                                        
                /* Find the cell in the refined grid (if it exists) */
                struct cell *c;
                find_top_cell(level, iX+i, iY+j, iZ+k, &c, N, domain);
                int exists = find_cell(level, iX+i, iY+j, iZ+k, &c);
                // printf("ere\n");
                if (exists) {
                    c->value += value / grid_cell_vol * (part_x*part_y*part_z);
                    printf("deposited %g at %g %g %g\n", value / grid_cell_vol * (part_x*part_y*part_z), c->x[0], c->x[1], c->x[2]);
                }
			}
		}
	}
    
}

int main() {
    
    printf("joehoe!\n");
    
    int N = 4;
    double BoxLen = 10.0;
    double DomainRes = BoxLen / N;
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
                
                printf ("%g %g %g\n", domain[row_major(x, y, z, N)].x[0], domain[row_major(x, y, z, N)].x[1], domain[row_major(x, y, z, N)].x[2]);
            }
        }
    }
    
    printf("=====\n");
    
    int refinement_memory = 1000;
    int refinement_counter = 0;
    struct cell *refinements = calloc(sizeof(struct cell), refinement_memory); 
    
    /* Refine the octree */
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            for (int z = 0; z < N; z++) {
                
                /* Refine? */
                int refine = (x % 2 == 1) * (y % 2 == 1) * (z % 3 == 1);
                
                if (refine) {
                    printf("%d %d %d\n", x, y, z);
                    
                    refine_cell(&domain[row_major(x, y, z, N)], refinements, &refinement_counter, refinement_memory);
                }
            }
        }
    }
    
    
    printf("=====\n");
    
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
    
    
    return 0;
}