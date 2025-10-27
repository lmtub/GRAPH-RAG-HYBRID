static struct pathelem *add_entry(struct pathelem *root, const char *name)

{

    root->num_entries++;



    root = realloc(root, sizeof(*root)

                   + sizeof(root->entries[0])*root->num_entries);



    root->entries[root->num_entries-1] = new_entry(root->pathname, root, name);

    root->entries[root->num_entries-1]

        = add_dir_maybe(root->entries[root->num_entries-1]);

    return root;

}
