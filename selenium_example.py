from selenium import webdriver
import common
from autotest_lib.client.common_lib.cros import chromedriver

with chromedriver.chromedriver() as chromedriver_instance:
   driver = chromedriver_instance.driver


#PATH = "/media/fuse/crostini_86e7e71744da6808e94e9f847bd470c173f254ed_termina_penguin/chromedriver"
#PATH = "/usr/local/chromedriver"
#driver = webdriver.Chrome(PATH)

URL = "https://techwithtim.net"
driver.get(URL)
