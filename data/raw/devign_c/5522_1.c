static int event_qdev_exit(DeviceState *qdev)

{

    SCLPEvent *event = DO_UPCAST(SCLPEvent, qdev, qdev);

    SCLPEventClass *child = SCLP_EVENT_GET_CLASS(event);

    if (child->exit) {

        child->exit(event);

    }

    return 0;

}
