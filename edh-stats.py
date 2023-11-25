import datetime
import json
import statistics
from collections import Counter

from trueskill import Rating, rate


class Deck:
    def __init__(self, name, architect):
        self.name = name + ' - ' + architect
        self.simple_name = name
        self.aliases = []
        self.architect = architect
        self.wins = 0
        self.played = 0
        self.score = 0
        self.matchups = {}
        self.wins_by_turn = Counter()
        self.eliminations = Counter()
        self.mmr = Rating()
        self._assists = 0
        self.games = []
        self.ranks = {
            'by_tbs': '',
            'by_delta': '',
            'by_wrx1': '',
            'by_mmr': '',
            'by_speed': ''
        }
        self.rank_delta = {
            'by_tbs': '',
            'by_delta': '',
            'by_wrx1': '',
            'by_mmr': '',
            'by_speed': ''
        }
        self.winrate = None
        self.expected_winrate = None
        self.winrate_delta = None
        self.op_winrate = None
        self.wrx1 = None
        self.mu = None
        self.sigma = None

        self._avg_win_turn = 100
        self._fastest_win = 100
        self._n_eliminations = 100
        self._avg_elim_turn = 100
        self._fastest_elim = 100

    def is_ranked(self):
        return self.played > 2

    def get_alias(self):
        if self.aliases:
            return self.aliases[0]
        else:
            return self.simple_name

    def get_assists(self):
        return self._assists or ''

    def get_avg_win_turn(self):
        if self.wins == 0:
            return 'n/a'
        elif len(self.wins_by_turn) == 0:
            return '?'
        else:
            return self._avg_win_turn

    def get_fastest_win(self):
        if self.wins == 0:
            return 'n/a'
        elif len(self.wins_by_turn) == 0:
            return '?'
        else:
            return self._fastest_win

    def get_n_eliminations(self):
        if len(self.eliminations) == 0:
            return '?'
        else:
            return self._n_eliminations

    def get_avg_elim_turn(self):
        if len(self.eliminations) == 0:
            return '?'
        else:
            return self._avg_elim_turn

    def get_fastest_elim(self):
        if len(self.eliminations) == 0:
            return '?'
        else:
            return self._fastest_elim

    def update_game_results(self, game):
        if self.name == game.winner:
            if game.elims_by_deck:
                points = sum(map(
                    len,
                    game.elims_by_deck[self.name].values()
                ))
            else:
                points = len(game.losers)
            self.score += points
            self.wins += 1
        else:
            self.score -= 1
        self.played += 1
        self.games.append(game)

    def update_matchups(self, game):
        if self.name == game.winner:
            for loser_name in game.losers:
                if loser_name not in self.matchups:
                    self.matchups[loser_name] = 0
                self.matchups[loser_name] += 1
        else:
            if game.winner not in self.matchups:
                self.matchups[game.winner] = 0
            self.matchups[game.winner] -= 1

    def update_eliminations(self, game):
        if not game.elims_by_deck:
            return

        elims_by_turn = game.elims_by_deck[self.name]

        if self.name == game.winner:
            turn_won = max(elims_by_turn.keys())
            self.wins_by_turn[turn_won] += 1
        for turn, losers in elims_by_turn.items():
            n_elims = len(losers)
            self.eliminations[turn] += n_elims
            if self.name != game.winner:
                self._assists += n_elims
                self.score += n_elims

    def update_rank(self, metric, rank):
        r = self.ranks[metric]
        if r:
            delta = r - rank
            if delta == 0:
                self.rank_delta[metric] = ''
            else:
                self.rank_delta[metric] = ('+' if delta > 0 else '') + str(delta)
        else:
            self.rank_delta[metric] = 'New!'
        self.ranks[metric] = rank

    def calculate_metrics(self, decks):
        avg_n_players = statistics.mean(
            map(lambda game: len(game.decks), self.games)
        )
        self.winrate = self.wins / self.played
        self.expected_winrate = 1 / avg_n_players
        self.winrate_delta = self.winrate - self.expected_winrate

        if len(self.wins_by_turn):
            self._avg_win_turn = statistics.mean(self.wins_by_turn.elements())
            self._fastest_win = min(self.wins_by_turn.keys())

        if len(self.eliminations):
            self._n_eliminations = self.eliminations.total()
            self._avg_elim_turn = statistics.mean(self.eliminations.elements())
            self._fastest_elim = min(self.eliminations.keys())

        op_wins = sum(map(lambda op: decks[op].wins, self.matchups))
        op_played = sum(map(lambda op: decks[op].played, self.matchups))
        self.op_winrate = op_wins / op_played

        self.wrx1 = (self.winrate + 1) * (self.op_winrate + 1)


class Game:
    def __init__(self, date, decks_by_turn_order, winner, archideck_names, eliminations):
        self.date = date
        self.decks = decks_by_turn_order
        self.winner = winner
        self.losers = [d for d in self.decks if d != self.winner]
        self.elims_by_deck = None
        self._rankings = [
            [winner],
            self.losers
        ]

        if not eliminations:
            return

        self.elims_by_deck = {
            deck_name: {}
            for deck_name in archideck_names.values()
        }

        assists_by_deck = Counter({
            deck: 0
            for deck in decks_by_turn_order
            if deck != winner
        })

        for t, es in eliminations.items():
            turn = int(t)
            for e in es:
                eliminator = archideck_names[e['eliminator']]
                eliminated = list(map(lambda l: archideck_names[l], e['eliminated']))
                self.elims_by_deck[eliminator][turn] = eliminated
                if eliminator != winner:
                    assists_by_deck[eliminator] += len(eliminated)

        if not assists_by_deck.total():
            return

        self._rankings = [[winner]]

        decks_by_assists = [[], [], []]
        for deck, assists in assists_by_deck.items():
            decks_by_assists[assists].append(deck)
        for decks in reversed(decks_by_assists):
            if len(decks):
                self._rankings.append(sorted(decks, key=decks_by_turn_order.index))

    def get_rankings(self):
        decks = []
        rankings = []
        for r, ds in enumerate(self._rankings):
            for d in ds:
                decks.append(d)
                rankings.append(r)
        return decks, rankings


def parse_records(filepath):
    decks = {}
    games = {}
    with open(filepath, 'r') as f:
        records = json.load(f)

    valid_players_keys = {'player', 'deck', 'architect', 'alias'}

    for record in records['games']:
        gs = record['games']
        date = record['date']
        games[date] = []

        for game in gs:
            players = game['players']
            winner = game['winner']
            simple_deck_names = list(map(lambda pxd: pxd['deck'], players))
            archideck_names = {}

            if winner not in simple_deck_names:
                raise Exception(f'winner {winner} not found in game dated {date}')

            for p in players:
                for key in p:
                    if key not in valid_players_keys:
                        raise Exception(f'invalid key {key} found in game dated {date}')
                player_name = p['player']
                simple_deck_name = p['deck']

                if 'architect' in p:
                    architect = p['architect']
                    if architect == player_name:
                        raise Exception(f'player {player_name} and architect match in game dated {date}')
                else:
                    architect = player_name

                archideck_name = simple_deck_name + ' - ' + architect
                archideck_names[simple_deck_name] = archideck_name
                if archideck_name not in decks:
                    if 'alias' in p:
                        alias_name = p['alias'] + ' - ' + architect
                        if alias_name not in decks:
                            raise Exception(f'alias {alias_name} for {archideck_name} not found in game dated {date}')
                        aliased_deck = decks[alias_name]
                        aliased_deck.aliases.insert(0, simple_deck_name)
                        archideck_names[simple_deck_name] = alias_name
                        decks[archideck_name] = aliased_deck
                    else:
                        decks[archideck_name] = Deck(simple_deck_name, architect)

            eliminations = {}

            if 'eliminations' in game:
                eliminations = game['eliminations']

                for t, es in eliminations.items():
                    for e in es:
                        eliminator = e['eliminator']
                        if eliminator not in simple_deck_names:
                            raise Exception(f'eliminator {eliminator} not found in game dated {date}')
                        eliminated = e['eliminated']
                        for loser in eliminated:
                            if loser not in simple_deck_names:
                                raise Exception(f'loser {loser} not found in game dated {date}')
                            if loser == winner:
                                raise Exception(f'winner {winner} eliminated in game dated {date}')

            game_object = Game(
                date,
                list(map(lambda sdn: archideck_names[sdn], simple_deck_names)),
                archideck_names[winner],
                archideck_names,
                eliminations
            )
            games[date].append(game_object)
    return decks, games


def update_record_results(decks, games):
    wins_by_turn_order = {
        3: Counter(),
        4: Counter()
    }

    games_as_list = [(d, gs) for d, gs in games.items()]

    for d, gs in games_as_list[:-1]:
        for game in gs:
            if parse_date(d) >= parse_date('2/2/2023'):
                wins_by_turn_order[len(game.decks)][game.decks.index(game.winner) + 1] += 1

            for deck_name in game.decks:
                deck = decks[deck_name]
                deck.update_game_results(game)
                deck.update_matchups(game)
                deck.update_eliminations(game)

            # deck_names, game_rankings = game.get_rankings()
            deck_names, _ = game.get_rankings()
            game_rankings = [0] + ([1] * len(game.losers))
            deck_ratings = [(decks[d].mmr,) for d in deck_names]
            new_ratings = rate(deck_ratings, ranks=game_rankings)
            for idx, (rating,) in enumerate(new_ratings):
                dd = deck_names[idx]
                ddd = decks[dd]
                ddd.mmr = rating

    last_game = games_as_list[-1]

    for deck in filter(lambda d: d.played, decks.values()):
        deck.calculate_metrics(decks)

    calculate_ranks(decks)

    for game in last_game[1]:
        wins_by_turn_order[len(game.decks)][game.decks.index(game.winner) + 1] += 1

        for deck_name in game.decks:
            deck = decks[deck_name]
            deck.update_game_results(game)
            deck.update_matchups(game)
            deck.update_eliminations(game)

        deck_names, game_rankings = game.get_rankings()
        deck_ratings = [(decks[d].mmr,) for d in deck_names]
        new_ratings = rate(deck_ratings, ranks=game_rankings)
        for idx, (rating,) in enumerate(new_ratings):
            decks[deck_names[idx]].mmr = rating

    for deck in decks.values():
        deck.calculate_metrics(decks)

    calculate_ranks(decks)

    return wins_by_turn_order


def by_speed(d):
    if type(d.get_avg_win_turn()) is str or type(d.get_avg_elim_turn()) is str:
        return 100, 100, 100, 100, 100

    return d.get_avg_win_turn(), d.get_avg_elim_turn(), d.get_fastest_win(), d.get_fastest_elim(), -d.winrate


def calculate_ranks(decks):
    ranked = list(filter(lambda d: d.is_ranked(), decks.values()))

    for metric, ranking in {
        'by_tbs': lambda d: (
                -d.winrate,
                -d.played,
                -d.op_winrate
        ),
        'by_delta': lambda d: (
                -d.winrate_delta,
                -d.played,
                -d.op_winrate
        ),
        'by_wrx1': lambda d: -d.wrx1,
        'by_mmr': lambda d: (
                -d.mmr.mu
        ),
        'by_speed': lambda d: (
                d._avg_win_turn,
                d._avg_elim_turn,
                d._fastest_win,
                d._fastest_elim,
                -d.winrate,
                -d._n_eliminations,
                -d.mmr.mu
        )
    }.items():
        for rank, deck in enumerate(sorted(ranked, key=ranking)):
            deck.update_rank(metric, rank+1)


def print_decks(decks, key):
    def print_deck(d):
        print(d.get_alias(), d.architect, d.wins, d.played, d.winrate, d.expected_winrate, d.winrate_delta,
              d.get_fastest_win(), d.get_avg_win_turn(), d.get_n_eliminations(), d.get_fastest_elim(),
              d.get_avg_elim_turn(), d.get_assists(), d.score, d.op_winrate, d.wrx1,
              d.mmr.mu, d.mmr.sigma,
              d.ranks['by_mmr'], d.rank_delta['by_mmr'],
              d.ranks['by_delta'], d.rank_delta['by_delta'],
              d.ranks['by_tbs'], d.rank_delta['by_tbs'],
              d.ranks['by_wrx1'], d.rank_delta['by_wrx1'],
              d.ranks['by_speed'], d.rank_delta['by_speed'],
              sep='\t'
              )

    header = "name\tplayer\twins\tplayed\twin %\tex. win %\twin % delta\tfastest win\tavg win turn\teliminations" \
             "\tfastest elimination\tavg elim turn\tassists\tscore\top win %" \
             "\twrx1 ((win % + 1) x (op win % + 1))\tmmr\tsigma" \
             "\tBy mmr\tΔ" \
             "\tBy tiebreaks (win % delta, played, op win %)\tΔ" \
             "\tBy tiebreaks (win %, played, op win %)\tΔ" \
             "\tBy wrx1\tΔ" \
             "\tBy speed\tΔ"
    print(header)

    aliased_decks = sorted({deck.get_alias(): deck for deck in decks.values()}.values(), key=key)

    ranked, unranked = partition(lambda d: d.is_ranked(),
                                 aliased_decks)
                                 # sorted({
                                 #     deck.get_alias(): deck
                                 #     for deck in decks.values()
                                 # },
                                 #     key=key))

    for deck in ranked:
        print_deck(deck)

    print()

    for deck in unranked:
        print_deck(deck)

    print()


def print_wins(wins):
    for p in [3, 4]:
        print(f'{p}-player games')
        print('player #\twins\twin%')
        total = sum(wins[p].values())
        for i in range(1, p+1):
            ws = wins[p][i]
            print(f'{i}\t{ws}\t{ws / total}')


def parse_date(d):
    m, d, y = [int(x) for x in d.split('/')]
    return datetime.date(y, m, d)


def partition(function, iterable):
    return filter(function, iterable), filter(lambda x: not function(x), iterable)


def main():
    by_score = lambda d: -d['score']
    alphabetical = lambda d: d.get_alias()

    record_filepath = './data/edh.json'

    date = "4/19/2999"
    decks, games = parse_records(record_filepath)
    wins = update_record_results(decks, games)
    print_decks(decks, key=alphabetical)
    print_wins(wins)


if __name__ == '__main__':
    main()
