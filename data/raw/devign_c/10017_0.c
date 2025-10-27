static void gui_update(void *opaque)

{

    uint64_t interval = GUI_REFRESH_INTERVAL;

    DisplayState *ds = opaque;

    DisplayChangeListener *dcl = ds->listeners;



    qemu_flush_coalesced_mmio_buffer();

    dpy_refresh(ds);



    while (dcl != NULL) {

        if (dcl->gui_timer_interval &&

            dcl->gui_timer_interval < interval)

            interval = dcl->gui_timer_interval;

        dcl = dcl->next;

    }

    qemu_mod_timer(ds->gui_timer, interval + qemu_get_clock(rt_clock));

}
