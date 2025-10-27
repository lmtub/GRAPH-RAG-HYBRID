static void do_safe_dpy_refresh(CPUState *cpu, run_on_cpu_data opaque)

{

    DisplayChangeListener *dcl = opaque.host_ptr;


    dcl->ops->dpy_refresh(dcl);


}