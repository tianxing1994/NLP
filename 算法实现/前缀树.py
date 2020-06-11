#!/usr/bin/python3
# -*- coding: utf-8 -*-


class Trie(object):
    def __init__(self):
        self.__is_word = "is_word"
        self.__root = {self.__is_word: False}

    def insert(self, word):
        """
        Insert a `word` to the trie.
        :param word: str
        :return: None.
        """
        now = self.__root
        for c in word:
            if c not in now:
                now[c] = {self.__is_word: False}
            now = now[c]
        now[self.__is_word] = True

    def search(self, word):
        """
        Return if the `word` in the trie,
        and the state of any pre-substring in the trie.
        pre-substring: means any substring start at the first character of
        `word`.

        :param word: str.
        :return: bool, list.

        example:
        >>> trie = Trie()
        >>> trie.insert('你好')
        >>> trie.insert('你好吗')
        >>> flag, state_list = trie.search('你好吗.')
        >>> print(flag, state_list)
        False [False, True, True, False]
        """
        state_list = list()
        now = self.__root
        for c in word:
            if c not in now:
                state_list.append(False)
                return False, state_list
            else:
                now = now[c]
                state_list.append(now[self.__is_word])
        return now[self.__is_word], state_list

    def starts_with(self, prefix, strict=False):
        """
        Return if there is any word starts with the `prefix`.
        :param prefix: str. the prefix to check.
        :param strict: bool. if True, the word same as prefix would not be accepted.
        :return: bool. is there any word starts with the `prefix`.
        """
        now = self.__root
        for c in prefix:
            if c not in now:
                return False
            else:
                now = now[c]
        if strict:
            return not now[self.__is_word]
        else:
            return True

    def get_words_by_prefix(self, prefix, strict=False):
        """
        Return all the words starts with the `prefix`.
        :param prefix: str. the prefix to check.
        :param strict: bool. if True, the word same as prefix would not be accepted.
        :return: list. all the words starts with the `prefix`.
        """
        words = list()

        # find the last node of `prefix`.
        now = self.__root
        for c in prefix:
            if c not in now:
                return list()
            else:
                now = now[c]

        # if `prefix` self is a word and not `strict`, add it.
        if now[self.__is_word] and not strict:
            words.append(prefix)

        # all sub tree is a task, add it to queue.
        queue = list()
        for k, v in now.items():
            if isinstance(v, bool):
                continue
            queue.append((prefix, k, v))

        # solve the queue.
        while len(queue) > 0:
            p, k, v = queue.pop()
            if v[self.__is_word]:
                word = p + k
                words.append(word)

            p += k
            for k, v in v.items():
                if isinstance(v, bool):
                    continue
                queue.append((p, k, v))
        return words


def demo1():
    trie = Trie()
    trie.insert('你好')
    trie.insert('你好吗')
    trie.insert('你好傻')
    trie.insert('你回家了吗')
    trie.insert('你是我的心')
    trie.insert('你是否对我也有情义')
    trie.insert('我心里都是你')
    trie.insert('我还是在想着你啊')

    flag, state_list = trie.search('你好吗.')
    print(flag, state_list)
    return


def demo2():
    trie = Trie()
    trie.insert('你好')
    trie.insert('你好吗')
    trie.insert('你好傻')
    trie.insert('你回家了吗')
    trie.insert('你是我的心')
    trie.insert('你是否对我也有情义')
    trie.insert('我心里都是你')
    trie.insert('我还是在想着你啊')

    # ret = trie.search('你是我的心')
    # print(ret)
    # ret = trie.starts_with(prefix='你')
    # print(ret)
    ret = trie.get_words_by_prefix(prefix='你好', strict=False)
    print(ret)
    return


if __name__ == '__main__':
    demo1()
    # demo2()
