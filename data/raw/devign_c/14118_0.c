static void spapr_vio_quiesce_one(VIOsPAPRDevice *dev)

{

    if (dev->tcet) {

        spapr_tce_reset(dev->tcet);

    }

    free_crq(dev);

}
