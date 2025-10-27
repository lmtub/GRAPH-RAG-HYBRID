static void xics_reset(DeviceState *d)

{

    XICSState *icp = XICS(d);

    int i;



    for (i = 0; i < icp->nr_servers; i++) {

        device_reset(DEVICE(&icp->ss[i]));

    }



    device_reset(DEVICE(icp->ics));

}
