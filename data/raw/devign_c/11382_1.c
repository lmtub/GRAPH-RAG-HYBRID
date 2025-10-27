void trace_event_set_vcpu_state_dynamic(CPUState *vcpu,

                                        TraceEvent *ev, bool state)

{

    TraceEventVCPUID vcpu_id;

    bool state_pre;

    assert(trace_event_get_state_static(ev));

    assert(trace_event_is_vcpu(ev));

    vcpu_id = trace_event_get_vcpu_id(ev);

    state_pre = test_bit(vcpu_id, vcpu->trace_dstate);

    if (state_pre != state) {

        if (state) {

            trace_events_enabled_count++;

            set_bit(vcpu_id, vcpu->trace_dstate);

            (*ev->dstate)++;

        } else {

            trace_events_enabled_count--;

            clear_bit(vcpu_id, vcpu->trace_dstate);

            (*ev->dstate)--;

        }

    }

}
