void hotplug_handler_post_plug(HotplugHandler *plug_handler,

                               DeviceState *plugged_dev,

                               Error **errp)

{

    HotplugHandlerClass *hdc = HOTPLUG_HANDLER_GET_CLASS(plug_handler);



    if (hdc->post_plug) {

        hdc->post_plug(plug_handler, plugged_dev, errp);

    }

}
