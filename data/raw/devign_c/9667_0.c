static void curses_calc_pad(void)

{

    if (is_graphic_console()) {

        width = gwidth;

        height = gheight;

    } else {

        width = COLS;

        height = LINES;

    }



    if (screenpad)

        delwin(screenpad);



    clear();

    refresh();



    screenpad = newpad(height, width);



    if (width > COLS) {

        px = (width - COLS) / 2;

        sminx = 0;

        smaxx = COLS;

    } else {

        px = 0;

        sminx = (COLS - width) / 2;

        smaxx = sminx + width;

    }



    if (height > LINES) {

        py = (height - LINES) / 2;

        sminy = 0;

        smaxy = LINES;

    } else {

        py = 0;

        sminy = (LINES - height) / 2;

        smaxy = sminy + height;

    }

}
