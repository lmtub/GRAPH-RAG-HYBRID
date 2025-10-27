void unregister_displaychangelistener(DisplayChangeListener *dcl)

{

    DisplayState *ds = dcl->ds;

    trace_displaychangelistener_unregister(dcl, dcl->ops->dpy_name);

    if (dcl->con) {

        dcl->con->dcls--;

    }

    QLIST_REMOVE(dcl, next);


    gui_setup_refresh(ds);

}