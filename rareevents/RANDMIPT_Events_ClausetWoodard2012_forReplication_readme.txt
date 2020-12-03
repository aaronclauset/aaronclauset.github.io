These event data are provided for replication purposes only, and were originally located at

   http://tuvalu.santafe.edu/~aaronc/rareevents/

Events were extracted from a 29 January 2008 full scrape of the MIPT database from www.tkb.org (defunct since Spring 2008). For more information about the RAND-MIPT data, see

   http://www.rand.org/nsrd/projects/terrorism-incidents.html (as of 2 January 2012)

or

   A. Clauset, M. Young and K. S. Gleditsch,
   "On the Frequency of Severe Terrorist Events."
   Journal of Conflict Resolution 51, 58 (2007).

If you use the RAND-MIPT data, cite them as

   National Memorial Institute for the Prevention of Terrorism, (2008)
   "Terrorism Knowledge Base."
   http://www.tkb.org (accessed 29 January 2008).

Finally, any questions regarding the collection protocol, quality of the data or access to the full database should be directed to the appropriate representatives at the RAND Corporation. As of June 2010, these were

   Andrew Morral (morral)
   Donald Temple (dtemple)

Email them at username@rand.org.

-Aaron Clauset (2 January 2012)

----------------------------------------
Description of file:

> record (integer)
Unique five-digit number indicating the RAND-MIPT record number of the event. Use this to join with other sources of data from RAND-MIPT.

> YYYY (integer)
Year of the event.

> deaths (integer)
Number of deaths. All events with deaths=0 have been dropped.

> weapon (integer)
Type of weapon used:
1 = biological or chemical agents
2 = explosives or remote detonated explosives
3 = fire, firebomb or arson
4 = firearms
5 = knives or sharp objects
6 = other or unknown

> OECD (binary)
1 = event occurred within an OECD nation (defined as the member nations  as of the 31 December 2007)
0 = elsewhere

> Iraq/Afghan (binary)
1 = event occurred within either Iraq or Afghanistan
0 = elsewhere
