from .chain import Chain
from .chain_4 import Chain4
from .chain_5 import Chain5
from .triangle import Triangle
from .collider import Collider
from .fork import Fork
from .diamond import Diamond
from .simpson import Simpson
from .large_backdoor import LargeBackdoor
from .german_credit import GermanCredit
from .simpson_wrong_1 import Simpson_wrong_1
from .simpson_wrong_2 import Simpson_wrong_2
sem_dict = {}

sem_dict["chain"] = Chain
sem_dict["chain-4"] = Chain4
sem_dict["chain-5"] = Chain5
sem_dict["triangle"] = Triangle
sem_dict["collider"] = Collider
sem_dict["fork"] = Fork
sem_dict["diamond"] = Diamond
sem_dict["simpson"] = Simpson
sem_dict["large-backdoor"] = LargeBackdoor
sem_dict["german"] = GermanCredit
sem_dict["simpson_wrong_1"]=Simpson_wrong_1
sem_dict["simpson_wrong_2"]=Simpson_wrong_2