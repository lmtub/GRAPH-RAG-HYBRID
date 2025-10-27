static void property_get_tm(Object *obj, Visitor *v, const char *name,

                            void *opaque, Error **errp)

{

    TMProperty *prop = opaque;

    Error *err = NULL;

    struct tm value;



    prop->get(obj, &value, &err);

    if (err) {

        goto out;

    }



    visit_start_struct(v, name, NULL, 0, &err);

    if (err) {

        goto out;

    }

    visit_type_int32(v, "tm_year", &value.tm_year, &err);

    if (err) {

        goto out_end;

    }

    visit_type_int32(v, "tm_mon", &value.tm_mon, &err);

    if (err) {

        goto out_end;

    }

    visit_type_int32(v, "tm_mday", &value.tm_mday, &err);

    if (err) {

        goto out_end;

    }

    visit_type_int32(v, "tm_hour", &value.tm_hour, &err);

    if (err) {

        goto out_end;

    }

    visit_type_int32(v, "tm_min", &value.tm_min, &err);

    if (err) {

        goto out_end;

    }

    visit_type_int32(v, "tm_sec", &value.tm_sec, &err);

    if (err) {

        goto out_end;

    }

out_end:

    error_propagate(errp, err);

    err = NULL;

    visit_end_struct(v, errp);

out:

    error_propagate(errp, err);



}
