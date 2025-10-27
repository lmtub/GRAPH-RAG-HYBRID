void xen_init_display(int domid)

{

    struct XenDevice *xfb, *xin;

    struct XenFB *fb;

    struct XenInput *in;

    int i = 0;



wait_more:

    i++;

    main_loop_wait(true);

    xfb = xen_be_find_xendev("vfb", domid, 0);

    xin = xen_be_find_xendev("vkbd", domid, 0);

    if (!xfb || !xin) {

        if (i < 256) {

            usleep(10000);

            goto wait_more;

        }

        xen_be_printf(NULL, 1, "displaystate setup failed\n");

        return;

    }



    /* vfb */

    fb = container_of(xfb, struct XenFB, c.xendev);

    fb->c.con = graphic_console_init(xenfb_update,

                                     xenfb_invalidate,

                                     NULL,

                                     NULL,

                                     fb);

    fb->have_console = 1;



    /* vkbd */

    in = container_of(xin, struct XenInput, c.xendev);

    in->c.con = fb->c.con;



    /* retry ->init() */

    xen_be_check_state(xin);

    xen_be_check_state(xfb);

}
