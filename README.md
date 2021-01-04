# messenger_method
This repository contains code used to take in CMB polarization maps and output B-mode only CMB maps. It does this through the use of a method dubbed the "messenger method" by Elsner and Wandelt (2012). To obtain B-mode purity, we have treated the E-mode part of the signal covariance as if it were noise covariance (in accordance with Bunn and Wandelt (2017)). 

There are two python files here. The first is messenger.py. This was the original version of the messenger method filter code. It's written in such a way that makes it very hard to understand/use. The code has since been rewritten in an object-oriented manner and this new version is in the file mod_mess.py. This code is easy to use and could easily be adapted to make an E-mode only filter as well, but it currently does not have that capability. Check out the Mod_mess Demonstration.ipynb file for a demonstration of how the mod_mess.py code works. More can be read about the applications of this code in the logbook posting and the poster linked to below. All of this code was written by me under the supervision of Dr. Colin Bischoff in the physics department at the University of Cincinnati.

Logbook posting: https://cmb-s4.org/wiki/index.php/PureB_by_Messenger_Method

Poster: https://journals.uc.edu/index.php/Undergradshowcase/article/view/4117/3124
