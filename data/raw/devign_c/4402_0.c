opts_start_optional(Visitor *v, bool *present, const char *name,

                       Error **errp)

{

    OptsVisitor *ov = DO_UPCAST(OptsVisitor, visitor, v);



    /* we only support a single mandatory scalar field in a list node */

    assert(ov->repeated_opts == NULL);

    *present = (lookup_distinct(ov, name, NULL) != NULL);

}
