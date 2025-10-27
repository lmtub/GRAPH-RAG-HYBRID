static sPAPRPHBState *find_phb(sPAPREnvironment *spapr, uint64_t buid)

{

    sPAPRPHBState *sphb;



    QLIST_FOREACH(sphb, &spapr->phbs, list) {

        if (sphb->buid != buid) {

            continue;

        }

        return sphb;

    }



    return NULL;

}
