static const HWAccel *get_hwaccel(enum AVPixelFormat pix_fmt, enum HWAccelID selected_hwaccel_id)

{

    int i;

    for (i = 0; hwaccels[i].name; i++)

        if (hwaccels[i].pix_fmt == pix_fmt &&

            (!selected_hwaccel_id || selected_hwaccel_id == HWACCEL_AUTO || hwaccels[i].id == selected_hwaccel_id))

            return &hwaccels[i];

    return NULL;

}
