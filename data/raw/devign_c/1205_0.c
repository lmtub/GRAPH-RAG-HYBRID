static inline uint16_t get_hwc_color(SM501State *state, int crt, int index)

{

    uint32_t color_reg = 0;

    uint16_t color_565 = 0;



    if (index == 0) {

        return 0;

    }



    switch (index) {

    case 1:

    case 2:

        color_reg = crt ? state->dc_crt_hwc_color_1_2

                        : state->dc_panel_hwc_color_1_2;

        break;

    case 3:

        color_reg = crt ? state->dc_crt_hwc_color_3

                        : state->dc_panel_hwc_color_3;

        break;

    default:

        printf("invalid hw cursor color.\n");

        abort();

    }



    switch (index) {

    case 1:

    case 3:

        color_565 = (uint16_t)(color_reg & 0xFFFF);

        break;

    case 2:

        color_565 = (uint16_t)((color_reg >> 16) & 0xFFFF);

        break;

    }

    return color_565;

}
