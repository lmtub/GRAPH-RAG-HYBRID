paint_mouse_pointer(XImage *image, struct x11_grab *s)

{

    int x_off = s->x_off;

    int y_off = s->y_off;

    int width = s->width;

    int height = s->height;

    Display *dpy = s->dpy;

    XFixesCursorImage *xcim;

    int x, y;

    int line, column;

    int to_line, to_column;

    int image_addr, xcim_addr;



    xcim = XFixesGetCursorImage(dpy);



    x = xcim->x - xcim->xhot;

    y = xcim->y - xcim->yhot;



    to_line = FFMIN((y + xcim->height), (height + y_off));

    to_column = FFMIN((x + xcim->width), (width + x_off));



    for (line = FFMAX(y, y_off); line < to_line; line++) {

        for (column = FFMAX(x, x_off); column < to_column; column++) {

            xcim_addr = (line - y) * xcim->width + column - x;



            if ((unsigned char)(xcim->pixels[xcim_addr] >> 24) != 0) { // skip fully transparent pixel

                image_addr = ((line - y_off) * width + column - x_off) * 4;



                image->data[image_addr] = (unsigned char)(xcim->pixels[xcim_addr] >> 0);

                image->data[image_addr+1] = (unsigned char)(xcim->pixels[xcim_addr] >> 8);

                image->data[image_addr+2] = (unsigned char)(xcim->pixels[xcim_addr] >> 16);

            }

        }

    }



    XFree(xcim);

    xcim = NULL;

}
