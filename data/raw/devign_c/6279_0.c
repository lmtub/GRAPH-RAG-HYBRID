void vnc_sent_lossy_rect(VncState *vs, int x, int y, int w, int h)

{

    int i, j;



    w = (x + w) / VNC_STAT_RECT;

    h = (y + h) / VNC_STAT_RECT;

    x /= VNC_STAT_RECT;

    y /= VNC_STAT_RECT;



    for (j = y; j <= y + h; j++) {

        for (i = x; i <= x + w; i++) {

            vs->lossy_rect[j][i] = 1;

        }

    }

}
