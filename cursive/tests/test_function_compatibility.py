def test_pydantic_compatibility():
    from cursive.function import cursive_function

    # Define a function with Pydantic v1 and v2 compatible annotations
    @cursive_function()
    def test_function(name: str, age: int):
        """
        A test function.
        Args:
            name: The name of a person.
            age: The age of the person.
        """
        return f"{name} is {age} years old."

    # Test the function with some arguments
    assert test_function("John", 30) == "John is 30 years old."

    # Check the function schema
    expected_schema = {
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The name of a person.", "title": "Name"},
                "age": {"type": "integer", "description": "The age of the person.", "title": "Age"},
            },
            "required": ["name", "age"],
        },
        "description": "A test function.\nArgs:\n    name: The name of a person.\n    age: The age of the person.",
        "name": "TestFunction",
    }
    assert test_function.function_schema == expected_schema
