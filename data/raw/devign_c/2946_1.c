static void paint_mouse_pointer(AVFormatContext *s1, struct gdigrab *gdigrab)

{

    CURSORINFO ci = {0};



#define CURSOR_ERROR(str)                 \

    if (!gdigrab->cursor_error_printed) {       \

        WIN32_API_ERROR(str);             \

        gdigrab->cursor_error_printed = 1;      \

    }



    ci.cbSize = sizeof(ci);



    if (GetCursorInfo(&ci)) {

        HCURSOR icon = CopyCursor(ci.hCursor);

        ICONINFO info;

        POINT pos;

        RECT clip_rect = gdigrab->clip_rect;

        HWND hwnd = gdigrab->hwnd;

        info.hbmMask = NULL;

        info.hbmColor = NULL;



        if (ci.flags != CURSOR_SHOWING)

            return;



        if (!icon) {

            /* Use the standard arrow cursor as a fallback.

             * You'll probably only hit this in Wine, which can't fetch

             * the current system cursor. */

            icon = CopyCursor(LoadCursor(NULL, IDC_ARROW));

        }



        if (!GetIconInfo(icon, &info)) {

            CURSOR_ERROR("Could not get icon info");

            goto icon_error;

        }



        pos.x = ci.ptScreenPos.x - clip_rect.left - info.xHotspot;

        pos.y = ci.ptScreenPos.y - clip_rect.top - info.yHotspot;



        if (hwnd) {

            RECT rect;



            if (GetWindowRect(hwnd, &rect)) {

                pos.x -= rect.left;

                pos.y -= rect.top;

            } else {

                CURSOR_ERROR("Couldn't get window rectangle");

                goto icon_error;

            }

        }



        av_log(s1, AV_LOG_DEBUG, "Cursor pos (%li,%li) -> (%li,%li)\n",

                ci.ptScreenPos.x, ci.ptScreenPos.y, pos.x, pos.y);



        if (pos.x >= 0 && pos.x <= clip_rect.right - clip_rect.left &&

                pos.y >= 0 && pos.y <= clip_rect.bottom - clip_rect.top) {

            if (!DrawIcon(gdigrab->dest_hdc, pos.x, pos.y, icon))

                CURSOR_ERROR("Couldn't draw icon");

        }



icon_error:





        if (icon)

            DestroyCursor(icon);

    } else {

        CURSOR_ERROR("Couldn't get cursor info");

    }

}