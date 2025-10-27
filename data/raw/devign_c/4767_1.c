static gpointer writeout_thread(gpointer opaque)

{

    TraceRecord *recordptr;

    union {

        TraceRecord rec;

        uint8_t bytes[sizeof(TraceRecord) + sizeof(uint64_t)];

    } dropped;

    unsigned int idx = 0;

    int dropped_count;

    size_t unused __attribute__ ((unused));



    for (;;) {

        wait_for_trace_records_available();



        if (g_atomic_int_get(&dropped_events)) {

            dropped.rec.event = DROPPED_EVENT_ID,

            dropped.rec.timestamp_ns = get_clock();

            dropped.rec.length = sizeof(TraceRecord) + sizeof(uint64_t),

            dropped.rec.reserved = 0;

            while (1) {

                dropped_count = g_atomic_int_get(&dropped_events);

                if (g_atomic_int_compare_and_exchange(&dropped_events,

                                                      dropped_count, 0)) {

                    break;

                }

            }

            dropped.rec.arguments[0] = dropped_count;

            unused = fwrite(&dropped.rec, dropped.rec.length, 1, trace_fp);

        }



        while (get_trace_record(idx, &recordptr)) {

            unused = fwrite(recordptr, recordptr->length, 1, trace_fp);

            writeout_idx += recordptr->length;

            free(recordptr); /* dont use g_free, can deadlock when traced */

            idx = writeout_idx % TRACE_BUF_LEN;

        }



        fflush(trace_fp);

    }

    return NULL;

}
