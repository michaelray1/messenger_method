# messenger_method
This repository contains code used to take in CMB polarization maps and output B-mode only CMB maps. It does this through the use of a method dubbed the Messenger
method by Elsner and Wandelt (2012). To obtain B-mode purity, we have treated the E-mode part of the signal covariance as if it were noise covariance (in
accordance with Bunn and Wandelt (2017)). Most of the files in here are old and unorganized. The files which contains the filter and is by far the most
readable and ready for reuse is mod_mess.py. This file contains all the functions needed to efficiently filter maps. The code could easily be adapted to 
make an E-mod only filter as well, but it currently does not have that capability. More can be read about the applications of this code in the logbook posting
and the poster linked to below. All of this code was written by me under the supervision of Dr. Colin Bischoff in the physics department at the University of
Cincinnati.

Logbook posting: https://cmb-s4.org/wiki/index.php/PureB_by_Messenger_Method
Poster: https://journals.uc.edu/index.php/Undergradshowcase/article/view/4117/3124
