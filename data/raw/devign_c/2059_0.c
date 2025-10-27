static struct vm_area_struct *vma_first(const struct mm_struct *mm)

{

    return (TAILQ_FIRST(&mm->mm_mmap));

}
