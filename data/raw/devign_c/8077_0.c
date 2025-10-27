ISADevice *pc_find_fdc0(void)

{

    int i;

    Object *container;

    CheckFdcState state = { 0 };



    for (i = 0; i < ARRAY_SIZE(fdc_container_path); i++) {

        container = container_get(qdev_get_machine(), fdc_container_path[i]);

        object_child_foreach(container, check_fdc, &state);

    }



    if (state.multiple) {

        error_report("warning: multiple floppy disk controllers with "

                     "iobase=0x3f0 have been found");

        error_printf("the one being picked for CMOS setup might not reflect "

                     "your intent\n");

    }



    return state.floppy;

}
