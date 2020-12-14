# messenger_method
This repository contains code used to take in CMB polarization maps and output B-mode only CMB maps. It does this through the use of a method dubbed the "messenger method" by Elsner and Wandelt (2012). To obtain B-mode purity, we have treated the E-mode part of the signal covariance as if it were noise covariance (in accordance with Bunn and Wandelt (2017)). Most of the files in here are old and unorganized. The file which contains the necessary functions for filtering is mod_mess.py. This file is also the most readable and by far the best code for reusability purposes in this repository. The code in this file could easily be adapted to make an E-mod only filter as well, but it currently does not have that capability. More can be read about the applications of this code in the logbook posting and the poster linked to below. All of this code was written by me under the supervision of Dr. Colin Bischoff in the physics department at the University of Cincinnati.

Logbook posting: https://cmb-s4.org/wiki/index.php/PureB_by_Messenger_Method

Poster: https://journals.uc.edu/index.php/Undergradshowcase/article/view/4117/3124
