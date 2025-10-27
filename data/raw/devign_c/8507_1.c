static void string_deserialize(void **native_out, void *datap,

                               VisitorFunc visit, Error **errp)

{

    StringSerializeData *d = datap;



    d->siv = string_input_visitor_new(string_output_get_string(d->sov));

    visit(string_input_get_visitor(d->siv), native_out, errp);

}
