static void qed_plug_allocating_write_reqs(BDRVQEDState *s)

{

    assert(!s->allocating_write_reqs_plugged);



    s->allocating_write_reqs_plugged = true;

}
