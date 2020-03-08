# chorus
Determine the bird from its call

## Resources

* Databases
    * https://www.macaulaylibrary.org
        * Tons of recordings, but requires signup/license agreement.
    * https://avocet.integrativebiology.natsci.msu.edu
    * https://www.xeno-canto.org

## Developing

This code requires python 3.7.

This project uses `pip-tools` to track requirements. It's recommended, but not required, to run the code.

```bash
# For running the code.
pip-sync requirements.txt
# alternatively, pip install -r requirements.txt

# To get packages that help with development as well:
pip-sync dev-requirements.txt

# If you just want to run tests, use the following.
pip-sync test-requirements.txt
```

If you want to try and be looser with the package versions, you can just reference the corresponding `.in` file. But be aware that tests are only run on the specific package versions referenced in the `.txt` files.
