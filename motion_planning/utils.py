VALID_COLOR_CHARS = {"r", "g", "b"}


def get_stacking_order_from_user() -> list[str]:
    """
    Prompts the user to enter the desired stacking order for the blocks.

    The input should be a 3-character string (e.g., 'rgb'), where:
    - The first character is the color of the base block.
    - The second character is the color of the block to be stacked first on the base.
    - The third character is the color of the block to be stacked on top of the second.

    Returns:
        (list[str]): A list of characters representing the colors of the blocks to be picked and
            stacked, in order.
    """
    while True:
        user_input = input(
            f"Enter the order of blocks (e.g., 'rgb', using {sorted(list(VALID_COLOR_CHARS))}): "
        ).lower()
        if len(user_input) == 3 and set(user_input) == VALID_COLOR_CHARS:
            pick_order_chars = [user_input[0], user_input[1], user_input[2]]
            print(
                f"\nUser-defined order: Base='{pick_order_chars[0].upper()}', "
                f"Stack 1='{pick_order_chars[1].upper()}', "
                f"Stack 2='{pick_order_chars[2].upper()}'"
            )
            return pick_order_chars
        else:
            print("Invalid input.")
