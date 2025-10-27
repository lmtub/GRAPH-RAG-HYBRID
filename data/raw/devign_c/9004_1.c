static uint16_t handle_write_event_buf(SCLPEventFacility *ef,

                                       EventBufferHeader *event_buf, SCCB *sccb)

{

    uint16_t rc;

    BusChild *kid;

    SCLPEvent *event;

    SCLPEventClass *ec;



    QTAILQ_FOREACH(kid, &ef->sbus.qbus.children, sibling) {

        DeviceState *qdev = kid->child;

        event = (SCLPEvent *) qdev;

        ec = SCLP_EVENT_GET_CLASS(event);



        rc = SCLP_RC_INVALID_FUNCTION;

        if (ec->write_event_data &&

            ec->event_type() == event_buf->type) {

            rc = ec->write_event_data(event, event_buf);

            break;

        }

    }

    return rc;

}
