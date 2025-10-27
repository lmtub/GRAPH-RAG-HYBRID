static int xcbgrab_reposition(AVFormatContext *s,

                              xcb_query_pointer_reply_t *p,

                              xcb_get_geometry_reply_t *geo)

{

    XCBGrabContext *c = s->priv_data;

    int x = c->x, y = c->y, p_x = p->win_x, p_y = p->win_y;

    int w = c->width, h = c->height, f = c->follow_mouse;



    if (!p || !geo)

        return AVERROR(EIO);



    if (f == FOLLOW_CENTER) {

        x = p_x - w / 2;

        y = p_y - h / 2;

    } else {

        int left   = x + f;

        int right  = x + w - f;

        int top    = y + f;

        int bottom = y + h + f;

        if (p_x > right) {

            x += p_x - right;

        } else if (p_x < left) {

            x -= left - p_x;

        }

        if (p_y > bottom) {

            y += p_y - bottom;

        } else if (p_y < top) {

            y -= top - p_y;

        }

    }



    c->x = FFMIN(FFMAX(0, x), geo->width  - w);

    c->y = FFMIN(FFMAX(0, y), geo->height - h);



    return 0;

}
