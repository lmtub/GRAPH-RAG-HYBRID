static int event_qdev_init(DeviceState *qdev)

{

    SCLPEvent *event = DO_UPCAST(SCLPEvent, qdev, qdev);

    SCLPEventClass *child = SCLP_EVENT_GET_CLASS(event);



    return child->init(event);

}
