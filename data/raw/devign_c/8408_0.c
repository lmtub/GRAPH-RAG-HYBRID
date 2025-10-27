VIOsPAPRDevice *spapr_vty_get_default(VIOsPAPRBus *bus)

{

    VIOsPAPRDevice *sdev, *selected;

    DeviceState *iter;



    /*

     * To avoid the console bouncing around we want one VTY to be

     * the "default". We haven't really got anything to go on, so

     * arbitrarily choose the one with the lowest reg value.

     */



    selected = NULL;

    QTAILQ_FOREACH(iter, &bus->bus.children, sibling) {

        /* Only look at VTY devices */

        if (qdev_get_info(iter) != &spapr_vty_info.qdev) {

            continue;

        }



        sdev = DO_UPCAST(VIOsPAPRDevice, qdev, iter);



        /* First VTY we've found, so it is selected for now */

        if (!selected) {

            selected = sdev;

            continue;

        }



        /* Choose VTY with lowest reg value */

        if (sdev->reg < selected->reg) {

            selected = sdev;

        }

    }



    return selected;

}
