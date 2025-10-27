static void visit_nested_struct(Visitor *v, void **native, Error **errp)

{

    visit_type_UserDefNested(v, (UserDefNested **)native, NULL, errp);

}
