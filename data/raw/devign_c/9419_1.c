void dpy_gl_scanout(QemuConsole *con,

                    uint32_t backing_id, bool backing_y_0_top,


                    uint32_t x, uint32_t y, uint32_t width, uint32_t height)

{

    assert(con->gl);

    con->gl->ops->dpy_gl_scanout(con->gl, backing_id,

                                 backing_y_0_top,


                                 x, y, width, height);

}