static void dpy_refresh(DisplayState *s)

{

    DisplayChangeListener *dcl;



    QLIST_FOREACH(dcl, &s->listeners, next) {

        if (dcl->ops->dpy_refresh) {

            dcl->ops->dpy_refresh(dcl);

        }

    }

}
