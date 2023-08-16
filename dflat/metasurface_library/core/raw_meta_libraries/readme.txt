Note that raw data files providing pre-computed optical responses for different nanolibraries can be included in this folder (e.g. .mat files)
Any file format may be provided since the libraryClass.py classes will incorporate corresponding calls to unpack the data when needed for training


PROPER USE FOR LIBRARY CLASSES:
- parameters of length are/should be provided in standard units of m
- parameters of angle are/should be provided in standard units of radians (ideally wrapped to 2pi but not necessary)
- transmission values are/should be unitless and between 0 and 1 
- it is good practice to make wavelength the last dimension since that dimension is frequently reduced in many cases


NOTES ON INCLUDED DATA FILES:

o data_Nanofins_Unit350nm_Height600nm_EngineFDTD.mat:
    - Generated via Lumerical FDTD parameter sweep on september 2021 (Dean Hazineh)
    - 350 nm unit cell size
    - response vs 1 nm step size in incident wavelength from 310 to 750 nm
    - response vs 5 nm step size in fin dimensions between 60 to 300 nm 
    - 
    - X and Y polarized plane wave sources simultaneously incident (injected inside glass susbtrate)
    - Bloch boundary conditions transverse to unit cell and PML normal
    - Record phase (minus reference field consisting of no TiO2 structure) at single point ~500um above structure 
    - Transmission manually computed by calculating x and y polarized light energy at single far away point ~500um above structure
        and normalizing by energy at that point with no structure present. 
        (The total light transmitted (polarization averaged) by this method is compared against the averaged Lumerical power monitor and shows good agreement between the two methods)

o data_Nanocylinders_Unit180nm_Height600nm_EngineFDTD.mat:
    - Generated via Lumerical FDTD parameter sweep on march 2022 (Dean Hazineh)
    - 180 nm unit cell size
    - response vs 1 nm step size in incident wavelength from 310 to 750 nm
    - response vs 1 nm step size in cylinder radius between 30 and 150 nm
    -
    - X and Y polarized plane wave sources simultaneously incident (injected inside glass susbtrate)
    - Bloch boundary conditions transverse to unit cell and PML normal
    - Record phase (minus reference field consisting of no TiO2 structure) at single point ~um above structure
        - phasex and phasey are equivalent so just phasex is uploaded in the mat file 
    - Transmission computed by Lumerical power monitor above nanostructure (lumerical built in call automatically normalizes to source power without any user work)

o data_NanoEllipse_Unit350nm_Height600nm_EngineFDTD.mat:
    - Generated via Lumerical FDTD parameter sweep on July 2022 (Dean Hazineh)
    - 350 nm unit cell seize
    - response vs 1 nm step size in incident wavelength from 310 to 750 nm
    - response vs 1 nm step size in cylinder radius between 60 and 300 nm
    -
    - X and Y polarized plane wave sources simultaneously incident (injected inside glass susbtrate)
    - Bloch boundary conditions transverse to unit cell and PML normal
    - Record phase (minus reference field consisting of no TiO2 structure) at single point ~500um above structure 
    - Transmission manually computed by calculating x and y polarized light energy at single far away point ~500um above structure
        and normalizing by energy at that point with no structure present. 
        (The total light transmitted (polarization averaged) by this method is compared against the averaged Lumerical power monitor and shows good agreement between the two methods)
