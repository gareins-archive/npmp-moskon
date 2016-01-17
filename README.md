# npmp-moskon

        # try:
        #     self.graph, info = odeint(self.diff_closure(), init, self.time, hmax=1, printmessg=False, full_output=True)
        #     if info['message'] != 'Integration successful.':
        #         return WORST_EVAL + 1
        #
        #     # HERE should go the evaluation of the result
        #     return self.evaluate_simple()
        #
        # except Exception:
        #     return WORST_EVAL + 2
