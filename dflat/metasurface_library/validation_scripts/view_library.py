from dflat.metasurface_library import *

if __name__ == "__main__":
    savefold = "dflat/metasurface_library/validation_scripts/output_library/"
    Nanofins_U350nm_H600nm().plotLibrary(savefold + "nanofins")
    Nanocylinders_U180nm_H600nm().plotLibrary(savefold + "nanocylinder")
    Nanoellipse_U350nm_H600nm().plotLibrary(savefold + "nanoellipse")
