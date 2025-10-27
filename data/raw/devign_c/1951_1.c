bool st_change_trace_event_state(const char *tname, bool tstate)

{

    TraceEvent *tp;



    tp = find_trace_event_by_name(tname);

    if (tp) {

        tp->state = tstate;

        return true;

    }

    return false;

}
