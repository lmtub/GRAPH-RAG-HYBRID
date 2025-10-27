int css_create_css_image(uint8_t cssid, bool default_image)

{

    trace_css_new_image(cssid, default_image ? "(default)" : "");

    if (cssid > MAX_CSSID) {

        return -EINVAL;

    }

    if (channel_subsys.css[cssid]) {

        return -EBUSY;

    }

    channel_subsys.css[cssid] = g_malloc0(sizeof(CssImage));

    if (default_image) {

        channel_subsys.default_cssid = cssid;

    }

    return 0;

}
