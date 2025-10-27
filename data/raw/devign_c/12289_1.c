TraceEvent *trace_event_iter_next(TraceEventIter *iter)

{

    while (iter->event < TRACE_EVENT_COUNT) {

        TraceEvent *ev = &(trace_events[iter->event]);

        iter->event++;

        if (!iter->pattern ||

            pattern_glob(iter->pattern,

                         trace_event_get_name(ev))) {

            return ev;

        }

    }



    return NULL;

}
