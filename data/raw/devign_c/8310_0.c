static enum AVHWDeviceType hw_device_match_type_by_hwaccel(enum HWAccelID hwaccel_id)

{

    int i;

    if (hwaccel_id == HWACCEL_NONE)

        return AV_HWDEVICE_TYPE_NONE;

    for (i = 0; hwaccels[i].name; i++) {

        if (hwaccels[i].id == hwaccel_id)

            return hwaccels[i].device_type;

    }

    return AV_HWDEVICE_TYPE_NONE;

}
