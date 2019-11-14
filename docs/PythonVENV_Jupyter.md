# Running Jupyter Notebook using python virtual environment

*David Lawrence Nov. 14, 2019*

This is how you can setup your JLab jupyterhub notebook to run inside a python virtual environment. This is useful for installing python packages via pip from your own account without needing admin privliges (which you don't have). The following will need to be run from a terminal on a CUE computer. I recommend doing this via a terminal launched via jupyterhub since you can use a python version available there that should have all necessary system libraries installed.

1. The JLab firewall will prevent pip from downloading packages using default options. The easiest thing to do is setup a pip configuration file in the home directory of your CUE account that sets the necessary options so you don't have to type them with every pip command. Create a file named "~/.config/pip/pip.conf" and write the following into it:

<pre>
<p style="background-color:lightgrey;">;
; Options to allow pip install to work from behind
; the hallgw firewall
;
   
[install]
trusted-host =
      pypi.org
      files.pythonhosted.org
  
[search]
trusted-host =
      pypi.org
      files.pythonhosted.org
</p>
</pre>

2. Create a python virtual environment in the directory of your choosing with:

   `python3.7 -m venv venv`


3. Source the new environment so it will be used for all subsequent pip commands:

   `source venv/bin/activate.csh`

  or

   `source venv/bin/activate.sh`


4. Upgrade the pip package itself to see that everything is working:

   `pip install --upgrade pip`


5. Install python packages you want:

   `pip install tensorflow keras pandas matplotlib imutils pillow scikit-learn opencv-python pypng`


6. Install the ipykernel into the venv and the venv into your list of kernels:

   `pip install ipykernel
python -m ipykernel install --user --name=venv`


7. At this point a new option for "venv" should be there in your launcher for both "Notebook" and "Console". You can verify that it is using your virtual environment by doing something like:

   `import sys
print(sys.path)`


Notes:

* The virtual environment is now part of a "kernel" and you can switch existing notebooks to using it via a menu in the upper right-hand corner of the notebook. (you may need to close and reopen the notebook to see the new kernel in the menu)
* You can list available kernels via command line using "**jupyter kernelspec list**"
* You can remove an old kernels using "**jupyter kernelspec uninstall old-kernel**"
