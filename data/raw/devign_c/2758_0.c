static void string_deserialize(void **native_out, void *datap,

                               VisitorFunc visit, Error **errp)

{

    StringSerializeData *d = datap;



    d->string = string_output_get_string(d->sov);

    d->siv = string_input_visitor_new(d->string);

    visit(d->siv, native_out, errp);

}
