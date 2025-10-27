void vnc_display_add_client(DisplayState *ds, int csock, int skipauth)

{

    VncDisplay *vs = ds ? (VncDisplay *)ds->opaque : vnc_display;



    vnc_connect(vs, csock, skipauth, 0);

}
