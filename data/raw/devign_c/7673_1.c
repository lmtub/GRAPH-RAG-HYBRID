static void setup_window(AVFormatContext *s)

{

    XCBGrabContext *c = s->priv_data;

    uint32_t mask     = XCB_CW_OVERRIDE_REDIRECT | XCB_CW_EVENT_MASK;

    uint32_t values[] = { 1,

                          XCB_EVENT_MASK_EXPOSURE |

                          XCB_EVENT_MASK_STRUCTURE_NOTIFY };

    xcb_rectangle_t rect = { 0, 0, c->width, c->height };



    c->window = xcb_generate_id(c->conn);



    xcb_create_window(c->conn, XCB_COPY_FROM_PARENT,

                      c->window,

                      c->screen->root,

                      c->x - c->region_border,

                      c->y - c->region_border,

                      c->width + c->region_border * 2,

                      c->height + c->region_border * 2,

                      0,

                      XCB_WINDOW_CLASS_INPUT_OUTPUT,

                      XCB_COPY_FROM_PARENT,

                      mask, values);



#if CONFIG_LIBXCB_SHAPE

    xcb_shape_rectangles(c->conn, XCB_SHAPE_SO_SUBTRACT,

                         XCB_SHAPE_SK_BOUNDING, XCB_CLIP_ORDERING_UNSORTED,

                         c->window,

                         c->region_border, c->region_border,

                         1, &rect);

#endif



    xcb_map_window(c->conn, c->window);



    draw_rectangle(s);

}
