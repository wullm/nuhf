/*******************************************************************************
 * This file is part of Mitos.
 * Copyright (c) 2020 Willem Elbers (whe@willemelbers.com)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <assert.h>
#include <math.h>
#include "../include/input.h"

int readParams(struct params *pars, const char *fname) {
     pars->GridSize = ini_getl("Box", "GridSize", 64, fname);
    
     pars->MaxParticleTypes = ini_getl("Simulation", "MaxParticleTypes", 1, fname);
     pars->NumParticleTypes = 0; //should not be read, but inferred
     pars->SlabSize = ini_getl("Read", "SlabSize", 8000000, fname);
    
     /* Read strings */
     int len = DEFAULT_STRING_LENGTH;
     pars->OutputDirectory = malloc(len);
     pars->InputFilename = malloc(len);
     pars->ImportName = malloc(len);
     pars->OutputFilename = malloc(len);
     ini_gets("Output", "Directory", "./output", pars->OutputDirectory, len, fname);
     ini_gets("Output", "Filename", "particles.hdf5", pars->OutputFilename, len, fname);
     ini_gets("Read", "Filename", "", pars->InputFilename, len, fname);
     ini_gets("Read", "ImportName", "", pars->ImportName, len, fname);

     return 0;
}

int readUnits(struct units *us, const char *fname) {
    /* Internal units */
    us->UnitLengthMetres = ini_getd("Units", "UnitLengthMetres", 1.0, fname);
    us->UnitTimeSeconds = ini_getd("Units", "UnitTimeSeconds", 1.0, fname);
    us->UnitMassKilogram = ini_getd("Units", "UnitMassKilogram", 1.0, fname);
    us->UnitTemperatureKelvin = ini_getd("Units", "UnitTemperatureKelvin", 1.0, fname);
    us->UnitCurrentAmpere = ini_getd("Units", "UnitCurrentAmpere", 1.0, fname);

    /* Get the transfer functions format */
    char format[DEFAULT_STRING_LENGTH];
    ini_gets("TransferFunctions", "Format", "Plain", format, DEFAULT_STRING_LENGTH, fname);

    /* Format of the transfer functions */
    int default_h_exponent;
    int default_k_exponent;
    int default_sign;
    if (strcmp(format, "CLASS") == 0) {
        default_h_exponent = 1;
        default_k_exponent = 0;
        default_sign = -1;
    } else {
        default_h_exponent = 0;
        default_k_exponent = -2;
        default_sign = +1;
    }
    us->TransferUnitLengthMetres = ini_getd("TransferFunctions", "UnitLengthMetres", MPC_METRES, fname);
    us->Transfer_hExponent = ini_getl("TransferFunctions", "hExponent", default_h_exponent, fname);
    us->Transfer_kExponent = ini_getl("TransferFunctions", "kExponent", default_k_exponent, fname);
    us->Transfer_Sign = ini_getl("TransferFunctions", "Sign", default_sign, fname);

    /* Some physical constants */
    us->SpeedOfLight = SPEED_OF_LIGHT_METRES_SECONDS * us->UnitTimeSeconds
                        / us->UnitLengthMetres;
    us->GravityG = GRAVITY_G_SI_UNITS * us->UnitTimeSeconds * us->UnitTimeSeconds
                    / us->UnitLengthMetres / us->UnitLengthMetres / us->UnitLengthMetres
                    * us->UnitMassKilogram; // m^3 / kg / s^2 to internal
    us->hPlanck = PLANCK_CONST_SI_UNITS / us->UnitMassKilogram / us->UnitLengthMetres
                    / us->UnitLengthMetres * us->UnitTimeSeconds; //J*s = kg*m^2/s
    us->kBoltzmann = BOLTZMANN_CONST_SI_UNITS / us->UnitMassKilogram / us->UnitLengthMetres
                    / us->UnitLengthMetres * us->UnitTimeSeconds * us->UnitTimeSeconds
                    * us->UnitTemperatureKelvin; //J/K = kg*m^2/s^2/K
    us->ElectronVolt = ELECTRONVOLT_SI_UNITS / us->UnitMassKilogram / us->UnitLengthMetres
                    / us->UnitLengthMetres * us->UnitTimeSeconds
                    * us->UnitTimeSeconds; // J = kg*m^2/s^2

    return 0;
}

int readCosmology(struct cosmology *cosmo, struct units *us, const char *fname) {
     cosmo->h = ini_getd("Cosmology", "h", 0.70, fname);
     cosmo->n_s = ini_getd("Cosmology", "n_s", 0.97, fname);
     cosmo->A_s = ini_getd("Cosmology", "A_s", 2.215e-9, fname);
     cosmo->k_pivot = ini_getd("Cosmology", "k_pivot", 0.05, fname);
     cosmo->z_ini = ini_getd("Cosmology", "z_ini", 40.0, fname);

     /* Default value for z_source is z_ini */
     cosmo->z_source = ini_getd("Cosmology", "z_source", cosmo->z_ini, fname);

     double H0 = 100 * cosmo->h * KM_METRES / MPC_METRES * us->UnitTimeSeconds;
     cosmo->rho_crit = 3 * H0 * H0 / (8 * M_PI * us->GravityG);

     return 0;
}

int cleanParams(struct params *pars) {
    free(pars->OutputDirectory);
    free(pars->InputFilename);
    free(pars->OutputFilename);
    free(pars->ImportName);

    return 0;
}